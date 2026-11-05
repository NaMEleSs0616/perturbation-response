# Benchmark results

All numbers are generated from run `bench-20261101-142417-9f7a7e` (seed 0, data version `931b2443c752beef`) unless stated. Results are in `results/norman2019/bench-20261101-142417-9f7a7e/`. Parentheses show 95% percentile bootstrap intervals over perturbations. No correction for multiple comparisons has been applied, and intervals are somewhat too narrow for pairs because pairs share genes.

```bash
python -m biophysical.cli benchmark run --config configs/benchmark_norman.yaml
python -m biophysical.cli benchmark representations --gene2vec data/external/gene2vec/gene2vec_dim_200_iter_9_w2v.txt --out results/norman2019/representations
```

## Summary

| Question | Answer on this dataset |
|---|---|
| Predict the response to a gene with no training data? | **No.** No model beats the mean of other perturbations (skill ≤ 0), though the measured responses are highly reproducible. The system abstains on this input by default. |
| Predict a pair when both singles are measured? | **Yes, largely by a simple rule.** The pair is close to 0.79 × (A + B). Learned models were not distinguishable from that one-parameter baseline on MSE. They gave a modest gain in specific correlation and in ranking pairs by size. |
| Predict the non-additive part of a pair? | **Only partly.** r = 0.44 against an estimated reproducibility limit of 0.75, but the zero and scaled-additive baselines already reach 0.37, because pairs are generally smaller than their sum. |
| Choose better experiments? | **For finding strong pairs, a measured-singles ranking gives about 2.6-fold enrichment,** and no learned model improved on it. Uncertainty-driven acquisition showed little practical benefit. |

## Design

- **Dataset:** Norman et al. 2019, K562 CRISPRa. 232 conditions (105 singles, 127 pairs), 2077 genes, 9702 control cells.
- **Target:** cell-weighted per-batch difference in mean log1p CP10k. Training uses control split 0, evaluation uses control split 1.
- **Splits (5-fold, grouped):** `unseen_gene` holds out all conditions containing a gene, with strata `single_unseen`, `combo_one_unseen`, `combo_two_unseen`. `unseen_combo` holds out pairs with both singles in training. Every fold passes `assert_no_leakage`. Assignments are in `split_assignments.csv`.
- **Same conditions for all:** all models in a fold are scored on the same perturbations and target.
- **Perturbed genes are masked** in their own conditions, in training and evaluation.

## Choice of metrics

The mean response over other perturbations already correlates at r ≈ 0.50 with a held-out single and r ≈ 0.59 with a held-out pair, because most perturbations share a common shift from controls. Raw correlation therefore overstates what a model has learned.

- **`skill_vs_mean`** is the primary metric: 1 − SSE(model) / SSE(training mean). Zero means no better than the mean.
- **`r, specific`** correlates each response's deviation from the training mean with the predicted deviation. The mean model has no value for it. It should not be read alone: the zero model scores 0.126 on singles, because a perturbation's deviation partly reflects its overall magnitude, so a model can have a positive value with no gene-specific knowledge. Skill is also needed.
- **`r, non-additive`** correlates (pair − A − B) with the predicted equivalent.

## Reproducibility limits

Mean of per-perturbation values. `max r` applies Spearman-Brown and approximates what a noiseless prediction could reach. It is an upper limit estimate, assuming independent noise, and is not a target that was validated independently.

| Stratum | Raw half-split r | Raw max r | Specific r, disjoint batches | Specific max r | Non-additive half-split r | Non-additive max r |
|---|---|---|---|---|---|---|
| combo_one_unseen | 0.872 | 0.960 | 0.841 | 0.953 | — | — |
| combo_seen_singles | 0.878 | 0.962 | 0.833 | 0.949 | 0.454 | 0.752 |
| combo_two_unseen | 0.900 | 0.972 | 0.846 | 0.950 | — | — |
| single_unseen | 0.806 | 0.935 | 0.793 | 0.932 | — | — |

Disjoint batches here means different 10x lanes of one experiment. This is a technical estimate; biological replicate variation is not included. 346 stratum-condition combinations have specific batch-disjoint r ≥ 0.5, so restricting to these (the `reproducible` subset in `summary.csv`) barely changes any result.

## Unseen single genes (n = 105)

| Model | Skill vs mean | r, specific | r, raw | MSE |
|---|---|---|---|---|
| `mean` | 0.000 (0.000, 0.000) | — | 0.504 (0.439, 0.562) | 0.004 (0.004, 0.005) |
| `gene_loading_singles_only` | -0.003 (-0.007, 0.001) | 0.028 (-0.024, 0.079) | 0.503 (0.438, 0.562) | 0.004 (0.004, 0.005) |
| `ridge_singles_only` | -0.003 (-0.010, 0.003) | 0.016 (-0.034, 0.066) | 0.506 (0.440, 0.564) | 0.004 (0.004, 0.005) |
| `gene_loading_ridge` | -0.018 (-0.037, -0.003) | 0.022 (-0.032, 0.078) | 0.499 (0.434, 0.558) | 0.004 (0.004, 0.005) |
| `ridge` | -0.124 (-0.234, -0.038) | 0.051 (-0.012, 0.110) | 0.493 (0.428, 0.549) | 0.005 (0.004, 0.005) |
| `bootstrap_ridge` | -0.127 (-0.226, -0.046) | 0.024 (-0.031, 0.083) | 0.488 (0.424, 0.545) | 0.005 (0.004, 0.005) |
| `mlp_ensemble` | -0.175 (-0.306, -0.062) | 0.062 (-0.002, 0.128) | 0.480 (0.413, 0.539) | 0.005 (0.004, 0.006) |
| `knn` | -0.288 (-0.372, -0.202) | -0.029 (-0.088, 0.031) | 0.462 (0.405, 0.513) | 0.005 (0.004, 0.006) |
| `zero` | -0.341 (-0.452, -0.220) | 0.126 (0.037, 0.221) | — | 0.006 (0.005, 0.007) |

No model has positive mean skill. `mlp_ensemble` vs `mean`, MSE difference: +0.00036 (+0.00012, +0.00062), Wilcoxon p = 0.012, n = 105. The singles-only and gene-loading models avoid harm only by collapsing to the mean. These results are far below the reproducibility limit, so noise does not explain the failure.

**Failure analysis**

- Damage is concentrated. Median skill of `mlp_ensemble` is -0.020, but its mean is -0.175. Worst: NCL (-4.79), LYL1 (-3.29), EGR1 (-2.34), CELF2 (-2.11), KIF2C (-2.03). Best: CBL (0.54), PTPN1 (0.52), ISL2 (0.36), ZNF318 (0.31), UBASH3A (0.26). Only 42% of singles have positive skill.
- Magnitude cannot be ranked either: Spearman across perturbations was -0.17 for `mlp_ensemble` and -0.09 for `gene_loading_ridge`.
- **Negative control:** shuffling the co-expression embedding among genes does not hurt (table below), so these features carry no detectable information about the unseen gene's response.
- The results are stable: skill stayed negative for all three models under all three other fold partitions.

This is consistent with published findings that mean and linear baselines are hard to beat for unseen perturbations (Ahlmann-Eltze, Huber & Anders, Nat Methods 2025). About 70 single-gene training conditions per fold, spanning diverse biology, may simply be too few. It is untested here whether larger screens change this.

## Pairs with both singles measured (n = 127)

| Model | Skill vs mean | r, specific | r, non-additive | Skill vs additive |
|---|---|---|---|---|
| `mlp_on_additive` | 0.629 (0.588, 0.666) | 0.832 (0.813, 0.850) | 0.442 (0.394, 0.488) | 0.180 (0.135, 0.225) |
| `ridge_on_additive` | 0.622 (0.581, 0.660) | 0.828 (0.809, 0.846) | 0.440 (0.400, 0.482) | 0.165 (0.118, 0.210) |
| `bootstrap_ridge_on_additive` | 0.615 (0.572, 0.653) | 0.825 (0.805, 0.843) | 0.432 (0.390, 0.472) | 0.136 (0.073, 0.191) |
| `scaled_additive` | 0.598 (0.555, 0.638) | 0.790 (0.767, 0.812) | 0.366 (0.301, 0.434) | 0.127 (0.073, 0.181) |
| `knn_on_additive` | 0.597 (0.539, 0.648) | 0.822 (0.799, 0.842) | 0.452 (0.409, 0.492) | 0.121 (0.054, 0.186) |
| `additive_plus_mean_residual` | 0.535 (0.478, 0.589) | 0.810 (0.787, 0.830) | 0.274 (0.237, 0.309) | 0.042 (0.014, 0.070) |
| `additive` | 0.485 (0.415, 0.548) | 0.796 (0.771, 0.818) | — | 0.000 (0.000, 0.000) |
| `mlp_ensemble` | 0.404 (0.348, 0.460) | 0.655 (0.611, 0.702) | 0.485 (0.447, 0.522) | -0.614 (-0.838, -0.416) |
| `ridge` | 0.222 (0.147, 0.289) | 0.516 (0.465, 0.565) | 0.431 (0.391, 0.470) | -1.177 (-1.515, -0.887) |
| `mean` | 0.000 (0.000, 0.000) | — | 0.426 (0.384, 0.467) | -2.134 (-2.585, -1.737) |
| `zero` | -0.655 (-0.782, -0.530) | 0.067 (-0.004, 0.136) | 0.366 (0.301, 0.434) | -3.390 (-3.831, -2.977) |

Paired comparisons of `mlp_on_additive`:

| Reference | MSE difference (model − reference) | Skill-vs-additive difference |
|---|---|---|
| `additive` | -0.00096 (-0.00127, -0.00067), Wilcoxon p = 1.6e-12, n = 127 | +0.180 (+0.135, +0.225) |
| `scaled_additive` | -0.00021 (-0.00043, +0.00004), Wilcoxon p = 0.084, n = 127 | +0.054 (-0.012, +0.117) |
| `additive_plus_mean_residual` | -0.00068 (-0.00092, -0.00046), Wilcoxon p = 2.8e-10, n = 127 | +0.138 (+0.098, +0.175) |
| `ridge_on_additive` | -0.00002 (-0.00009, +0.00006), Wilcoxon p = 0.096, n = 127 | +0.016 (-0.005, +0.037) |

- **Most of the gain over additive is saturation.** The fitted scale is 0.78 to 0.80 across folds, and that single number gives most of the improvement. The learned models' MSE advantage over it has an interval that includes zero on this partition. Across partitions, `mlp_on_additive` skill stayed above `scaled_additive` (table below), a consistent but small and non-independent difference.
- **Where learned models do differ:** `r, specific` (+0.042 (+0.028, +0.056) vs scaled additive) and magnitude ranking (Spearman 0.80 vs 0.69 for additive).
- **MLP vs ridge:** not distinguishable, so the neural model adds no demonstrated value over ridge on the same inputs.
- Largest failures for `mlp_on_additive` (skill vs additive): MAP2K3_SLC38A2 (-0.71), MAP2K3_MAP2K6 (-0.51), DUSP9_KLF1 (-0.37), MAPK1_TGFBR2 (-0.36). Full table: `per_perturbation_metrics.csv`.

Pairs in this screen were chosen by the original authors, so this may not transfer to arbitrary pairs.

## Pairs with an unmeasured gene

One gene unseen (n = 100):

| Model | Skill vs mean | r, specific | r, raw | MSE |
|---|---|---|---|---|
| `mlp_ensemble` | 0.071 (0.033, 0.108) | 0.309 (0.261, 0.360) | 0.612 (0.560, 0.662) | 0.007 (0.007, 0.008) |
| `ridge` | 0.051 (0.018, 0.082) | 0.291 (0.232, 0.354) | 0.604 (0.550, 0.653) | 0.008 (0.007, 0.008) |
| `bootstrap_ridge` | 0.042 (0.011, 0.073) | 0.268 (0.208, 0.329) | 0.600 (0.547, 0.649) | 0.008 (0.007, 0.009) |
| `mean` | 0.000 (0.000, 0.000) | — | 0.568 (0.516, 0.615) | 0.008 (0.007, 0.009) |
| `knn` | -0.031 (-0.101, 0.040) | 0.240 (0.183, 0.299) | 0.601 (0.552, 0.648) | 0.008 (0.007, 0.009) |

Both genes unseen (n = 27):

| Model | Skill vs mean | r, specific | r, raw | MSE |
|---|---|---|---|---|
| `mlp_ensemble` | 0.059 (-0.002, 0.122) | 0.278 (0.167, 0.393) | 0.594 (0.502, 0.664) | 0.009 (0.007, 0.011) |
| `ridge` | 0.038 (-0.043, 0.125) | 0.219 (0.097, 0.352) | 0.592 (0.500, 0.665) | 0.009 (0.007, 0.011) |
| `mean` | 0.000 (0.000, 0.000) | — | 0.555 (0.464, 0.626) | 0.010 (0.008, 0.012) |
| `knn` | -0.079 (-0.215, 0.050) | 0.159 (0.025, 0.292) | 0.571 (0.488, 0.641) | 0.010 (0.008, 0.012) |

Skill is small, and with both genes unseen its interval includes zero. These are evaluated against the mean; no additive prediction is possible.

## Uncertainty

| Stratum | Model | Coverage, all genes (target 0.90) | Coverage, top 20 responders | Disagreement vs RMSE (Spearman) |
|---|---|---|---|---|
| single_unseen | `mlp_ensemble` | 0.914 | 0.565 | 0.047 (p = 0.63) |
| combo_seen_singles | `mlp_on_additive` | 0.863 | 0.742 | 0.211 (p = 0.017) |
| combo_seen_singles | `bootstrap_ridge_on_additive` | 0.863 | 0.740 | 0.340 (p = 9.3e-05) |
| combo_one_unseen | `mlp_ensemble` | 0.841 | 0.425 | 0.017 (p = 0.81) |
| combo_two_unseen | `mlp_ensemble` | 0.804 | 0.330 | -0.356 (p = 0.068) |

- **Intervals are not reliable for strong responders.** Overall coverage is near target but misses most on the genes that matter. Coverage on the top 20 genes falls well below the 0.90 target. Prediction records state this.
- **Disagreement ranks error only weakly.** It has no detectable relation to error for unseen genes, where it would be most needed, and a modest one for pairs.
- Interval widths are per gene and do not vary by perturbation, so they say nothing about which perturbation is harder.

## Choosing experiments

**Finding strong pairs out of sample** (n = 127, top 20%, random precision = 0.20):

| Ranking | Precision in top 20% | Fold enrichment over random | 95% interval | Spearman |
|---|---|---|---|---|
| `additive` | 0.52 | 2.6 | 0.36–0.68 | 0.69 |
| `scaled_additive` | 0.52 | 2.6 | 0.32–0.68 | 0.68 |
| `mlp_on_additive` | 0.52 | 2.6 | 0.32–0.68 | 0.80 |
| `ridge_on_additive` | 0.48 | 2.4 | 0.32–0.68 | 0.80 |
| `additive_plus_mean_residual` | 0.56 | 2.8 | 0.36–0.68 | 0.69 |
| `mlp_ensemble` | 0.48 | 2.4 | 0.24–0.60 | 0.51 |
| `mean` | 0.20 | 1.0 | 0.04–0.36 | -0.21 |

**Finding the most non-additive pairs:**

| Ranking | Precision in top 20% | Fold enrichment over random | 95% interval | Spearman |
|---|---|---|---|---|
| `additive_magnitude` | 0.56 | 2.8 | 0.36–0.72 | 0.62 |
| `scaled_additive` | 0.56 | 2.8 | 0.32–0.72 | 0.62 |
| `ridge_on_additive` | 0.52 | 2.6 | 0.36–0.72 | 0.49 |
| `mlp_on_additive` | 0.48 | 2.4 | 0.32–0.68 | 0.37 |
| `mean` | 0.56 | 2.8 | 0.32–0.68 | 0.38 |

Non-additive magnitude here is measured with noise and tracks how large the singles are, so "non-additive" hits are not necessarily synergistic biology. No learned ranking beat the summed magnitude of singles.

**Finding strong unseen singles** (n = 105):

| Ranking | Precision in top 20% | Fold enrichment over random | 95% interval | Spearman |
|---|---|---|---|---|
| `mlp_ensemble` | 0.14 | 0.7 | 0.00–0.29 | -0.17 |
| `ridge` | 0.19 | 1.0 | 0.05–0.33 | -0.09 |
| `gene_loading_ridge` | 0.24 | 1.2 | 0.05–0.38 | -0.09 |
| `mean` | 0.14 | 0.7 | 0.00–0.33 | -0.11 |

No better than random.

**Sequential acquisition, skill vs mean** (10 repeats, bootstrap ridge ensemble, same seed set per repeat):

Choosing pairs when singles are known:

| Strategy | Mean AUC | Difference from random (95% interval) | Repeats better than random |
|---|---|---|---|
| uncertainty | 0.5624 | +0.0082 (-0.0007, +0.0175) | 5/10 |
| hybrid | 0.5619 | +0.0077 (+0.0005, +0.0155) | 6/10 |
| diversity | 0.5601 | +0.0059 (+0.0005, +0.0126) | 8/10 |
| random | 0.5542 | +0.0000 (+0.0000, +0.0000) | 0/10 |

Choosing single genes:

| Strategy | Mean AUC | Difference from random (95% interval) | Repeats better than random |
|---|---|---|---|
| hybrid | -0.0855 | +0.0442 (+0.0126, +0.0775) | 7/10 |
| uncertainty | -0.0870 | +0.0428 (+0.0139, +0.0743) | 8/10 |
| diversity | -0.0942 | +0.0355 (+0.0056, +0.0682) | 5/10 |
| random | -0.1297 | +0.0000 (+0.0000, +0.0000) | 0/10 |

Differences are small. For singles, the model still has negative skill after acquisition, so a better training set only reduces harm. With 10 repeats and no multiplicity correction, these do not justify a claim that any strategy is better in practice. Expected improvement and magnitude are implemented but were not evaluated sequentially.

## Ablations (unseen singles unless noted)

**Representation (negative control included):**

| Variant | Model | Skill vs mean | r, specific | r, raw | MSE |
|---|---|---|---|---|---|
| coexpression | `mlp_ensemble` | -0.175 | 0.062 | 0.480 | 0.0047 |
| coexpression | `ridge` | -0.124 | 0.051 | 0.493 | 0.0045 |
| baseline_only | `mlp_ensemble` | -0.019 | 0.035 | 0.493 | 0.0044 |
| baseline_only | `ridge` | -0.033 | 0.014 | 0.501 | 0.0043 |
| permuted_coexpression | `mlp_ensemble` | -0.126 | -0.009 | 0.467 | 0.0045 |
| permuted_coexpression | `ridge` | -0.077 | -0.043 | 0.482 | 0.0044 |

**Shared vs disjoint controls:** using disjoint controls lowers the mean baseline's raw r only slightly. The effect in this dataset, with about 9,700 control cells, is small, though the design removes the bias by construction.

| Variant | Model | Skill vs mean | r, specific | r, raw | MSE |
|---|---|---|---|---|---|
| disjoint | `mean` | 0.000 | — | 0.504 | 0.0043 |
| disjoint | `ridge` | -0.124 | 0.051 | 0.493 | 0.0045 |
| shared | `mean` | 0.000 | — | 0.512 | 0.0043 |
| shared | `ridge` | -0.126 | 0.052 | 0.501 | 0.0045 |

**Target definition and preprocessing.** Absolute values differ with the gene set, so compare models within a row group. The conclusion that nothing beats the mean did not change under any variant.

| Variant | Model | Skill vs mean | r, specific | r, raw | MSE |
|---|---|---|---|---|---|
| log_of_pseudobulk_mean_shared_controls | `mean` | 0.000 | — | 0.478 | 0.0055 |
| log_of_pseudobulk_mean_shared_controls | `ridge` | -0.160 | 0.056 | 0.459 | 0.0058 |
| pseudobulk_mean_of_log_shared_controls | `mean` | 0.000 | — | 0.512 | 0.0043 |
| pseudobulk_mean_of_log_shared_controls | `ridge` | -0.126 | 0.052 | 0.501 | 0.0045 |

| Variant | Model | Skill vs mean | r, specific | r, raw | MSE |
|---|---|---|---|---|---|
| no_cell_filter | `gene_loading_ridge` | -0.015 | 0.027 | 0.499 | 0.0037 |
| no_cell_filter | `mean` | 0.000 | — | 0.503 | 0.0037 |
| no_cell_filter | `ridge` | -0.168 | 0.055 | 0.475 | 0.0039 |
| hvg_1000 | `gene_loading_ridge` | -0.019 | 0.019 | 0.485 | 0.0059 |
| hvg_1000 | `mean` | 0.000 | — | 0.489 | 0.0059 |
| hvg_1000 | `ridge` | -0.153 | 0.056 | 0.472 | 0.0061 |
| hvg_5000 | `gene_loading_ridge` | -0.017 | 0.025 | 0.505 | 0.0026 |
| hvg_5000 | `mean` | 0.000 | — | 0.509 | 0.0026 |
| hvg_5000 | `ridge` | -0.104 | 0.016 | 0.485 | 0.0027 |
| top_expressed_2000 | `gene_loading_ridge` | -0.021 | 0.015 | 0.503 | 0.0077 |
| top_expressed_2000 | `mean` | 0.000 | — | 0.509 | 0.0076 |
| top_expressed_2000 | `ridge` | -0.129 | 0.050 | 0.488 | 0.0080 |

**Other fold partitions (skill vs mean):**

| Stratum | Model | Partition 1 | Partition 2 | Partition 3 |
|---|---|---|---|---|
| combo_seen_singles | `additive` | 0.488 | 0.488 | 0.481 |
| combo_seen_singles | `mlp_on_additive` | 0.622 | 0.615 | 0.629 |
| combo_seen_singles | `scaled_additive` | 0.592 | 0.603 | 0.595 |
| single_unseen | `gene_loading_ridge` | -0.135 | -0.048 | -0.329 |
| single_unseen | `mlp_ensemble` | -0.224 | -0.124 | -0.151 |
| single_unseen | `ridge` | -0.281 | -0.384 | -0.080 |

## External representations

From `results/norman2019/representations/`. Skill differences are paired per perturbation against co-expression.

Network: STRING 12.5. 1730 of 2077 genes mapped; 24467 edges at combined score ≥ 400, 18013 with evidence from experimental or curated channels only. Median target degree 16; 3 targets isolated. Gene2vec: 1830 of 2077 genes have vectors; targets without one are CBARP, FOXL2NB, RP5-862P8.2.

Unseen singles:

| Representation | Model | Skill vs mean | r, specific | Skill difference vs co-expression (95% interval) |
|---|---|---|---|---|
| coexpression | `ridge` | -0.124 | 0.051 | reference |
| coexpression | `mlp_ensemble` | -0.175 | 0.062 | reference |
| baseline_only | `ridge` | -0.033 | 0.014 | +0.091 (-0.022, +0.210) |
| baseline_only | `mlp_ensemble` | -0.019 | 0.035 | +0.157 (+0.039, +0.294) |
| permuted_coexpression | `ridge` | -0.077 | -0.043 | +0.048 (-0.053, +0.165) |
| permuted_coexpression | `mlp_ensemble` | -0.126 | -0.009 | +0.049 (-0.090, +0.209) |
| gene2vec_frozen | `ridge` | -0.194 | 0.121 | -0.070 (-0.256, +0.096) |
| gene2vec_frozen | `mlp_ensemble` | -0.151 | 0.131 | +0.024 (-0.222, +0.237) |
| coexpr+gene2vec_frozen | `ridge` | -0.180 | 0.125 | -0.056 (-0.207, +0.078) |
| coexpr+gene2vec_frozen | `mlp_ensemble` | -0.142 | 0.125 | +0.034 (-0.193, +0.225) |
| coexpr+string_neighbours_combined | `ridge` | -0.168 | 0.081 | -0.044 (-0.127, +0.053) |
| coexpr+string_neighbours_combined | `mlp_ensemble` | -0.223 | 0.082 | -0.048 (-0.171, +0.079) |
| coexpr+string_neighbours_experimental_curated | `ridge` | -0.303 | 0.080 | -0.179 (-0.293, -0.079) |
| coexpr+string_neighbours_experimental_curated | `mlp_ensemble` | -0.207 | 0.106 | -0.032 (-0.114, +0.063) |
| string_spectral_combined | `ridge` | -0.172 | 0.163 | -0.047 (-0.156, +0.071) |
| string_spectral_combined | `mlp_ensemble` | -0.077 | 0.176 | +0.099 (-0.008, +0.231) |
| string_spectral_experimental_curated | `ridge` | -0.423 | 0.137 | -0.298 (-0.936, +0.105) |
| string_spectral_experimental_curated | `mlp_ensemble` | -0.157 | 0.148 | +0.018 (-0.320, +0.265) |

One gene unseen in a pair:

| Representation | Model | Skill vs mean | r, specific | Skill difference vs co-expression (95% interval) |
|---|---|---|---|---|
| coexpression | `ridge` | 0.051 | 0.291 | reference |
| coexpression | `mlp_ensemble` | 0.071 | 0.309 | reference |
| baseline_only | `ridge` | -0.000 | 0.030 | -0.051 (-0.082, -0.018) |
| baseline_only | `mlp_ensemble` | -0.009 | 0.062 | -0.080 (-0.115, -0.042) |
| permuted_coexpression | `ridge` | 0.021 | 0.204 | -0.030 (-0.067, +0.010) |
| permuted_coexpression | `mlp_ensemble` | 0.061 | 0.230 | -0.010 (-0.048, +0.031) |
| gene2vec_frozen | `ridge` | 0.132 | 0.440 | +0.081 (+0.025, +0.137) |
| gene2vec_frozen | `mlp_ensemble` | 0.202 | 0.470 | +0.131 (+0.083, +0.179) |
| coexpr+gene2vec_frozen | `ridge` | 0.150 | 0.452 | +0.099 (+0.045, +0.152) |
| coexpr+gene2vec_frozen | `mlp_ensemble` | 0.206 | 0.474 | +0.135 (+0.087, +0.184) |
| coexpr+string_neighbours_combined | `ridge` | 0.068 | 0.344 | +0.017 (-0.022, +0.057) |
| coexpr+string_neighbours_combined | `mlp_ensemble` | 0.090 | 0.395 | +0.019 (-0.039, +0.071) |
| coexpr+string_neighbours_experimental_curated | `ridge` | 0.086 | 0.371 | +0.035 (-0.009, +0.078) |
| coexpr+string_neighbours_experimental_curated | `mlp_ensemble` | 0.132 | 0.408 | +0.062 (+0.023, +0.105) |
| string_spectral_combined | `ridge` | 0.073 | 0.324 | +0.022 (-0.024, +0.070) |
| string_spectral_combined | `mlp_ensemble` | 0.117 | 0.360 | +0.046 (-0.001, +0.100) |
| string_spectral_experimental_curated | `ridge` | -0.015 | 0.314 | -0.066 (-0.248, +0.042) |
| string_spectral_experimental_curated | `mlp_ensemble` | 0.036 | 0.307 | -0.035 (-0.129, +0.037) |

Both singles measured:

| Representation | Model | Skill vs mean | r, specific | Skill difference vs co-expression (95% interval) |
|---|---|---|---|---|
| coexpression | `ridge_on_additive` | 0.622 | 0.828 | reference |
| coexpression | `mlp_on_additive` | 0.629 | 0.832 | reference |
| baseline_only | `ridge_on_additive` | 0.535 | 0.811 | -0.087 (-0.119, -0.057) |
| baseline_only | `mlp_on_additive` | 0.524 | 0.808 | -0.105 (-0.144, -0.069) |
| permuted_coexpression | `ridge_on_additive` | 0.582 | 0.813 | -0.040 (-0.059, -0.022) |
| permuted_coexpression | `mlp_on_additive` | 0.591 | 0.818 | -0.038 (-0.056, -0.020) |
| gene2vec_frozen | `ridge_on_additive` | 0.646 | 0.833 | +0.024 (+0.007, +0.040) |
| gene2vec_frozen | `mlp_on_additive` | 0.633 | 0.831 | +0.005 (-0.008, +0.018) |
| coexpr+gene2vec_frozen | `ridge_on_additive` | 0.648 | 0.833 | +0.025 (+0.008, +0.041) |
| coexpr+gene2vec_frozen | `mlp_on_additive` | 0.636 | 0.833 | +0.007 (-0.004, +0.018) |
| coexpr+string_neighbours_combined | `ridge_on_additive` | 0.633 | 0.830 | +0.011 (-0.004, +0.025) |
| coexpr+string_neighbours_combined | `mlp_on_additive` | 0.621 | 0.829 | -0.008 (-0.022, +0.005) |
| coexpr+string_neighbours_experimental_curated | `ridge_on_additive` | 0.639 | 0.831 | +0.017 (+0.005, +0.027) |
| coexpr+string_neighbours_experimental_curated | `mlp_on_additive` | 0.628 | 0.829 | -0.001 (-0.013, +0.012) |
| string_spectral_combined | `ridge_on_additive` | 0.580 | 0.812 | -0.042 (-0.075, -0.012) |
| string_spectral_combined | `mlp_on_additive` | 0.583 | 0.819 | -0.046 (-0.080, -0.016) |
| string_spectral_experimental_curated | `ridge_on_additive` | 0.586 | 0.817 | -0.036 (-0.068, -0.008) |
| string_spectral_experimental_curated | `mlp_on_additive` | 0.573 | 0.817 | -0.055 (-0.090, -0.025) |

- **Unseen singles:** no representation gives positive skill. Gene2vec and STRING spectral features raise specific correlation (to about 0.12–0.18), but their skill intervals against co-expression all include zero, so this is not a demonstrated improvement. Baseline-only features with the MLP do significantly less harm than co-expression, which shows overfitting to uninformative features rather than a useful signal.
- **One unseen gene in a pair:** frozen Gene2vec gives the clearest gain seen in this project (+0.131 skill vs co-expression for the MLP). It was not pre-specified, so it is a hypothesis to confirm on a separate screen.
- **STRING:** no consistent improvement over co-expression. The one interval above zero for unseen pairs is experimental and curated neighbour features with the MLP, and the same features with ridge do not reproduce it. On unseen singles those features significantly hurt ridge. This tests features derived from the graph and does not rule out other graph models.
- **Numerical failure, fixed:** the first spectral feature run gave skill as low as −21.8 for `string_spectral_experimental_curated`, because eigenvectors localised on tiny graph components. Standardising and clipping fixed this. That first run is recorded here because a failure of this kind would otherwise be invisible in a summary table.

## Runtime and memory

Apple M3, 8 GB, CPU only. Benchmark with cached preprocessing: 114 s, peak RSS 542 MB, including simulation and split-seed ablations. Preprocessing the full file: about 60 s, peak 1.3 GB. Mean model fit time per fold: MLP ensemble 0.83 s, ridge 0.008 s, additive 0.001 s.

Exact rerun: two consecutive complete runs gave identical values on all 5,020 per-perturbation rows (maximum absolute difference 0 for MSE, Pearson, skill and coverage). That was on one machine and library set; other platforms may differ.

## Process disclosure

Scaled additive, additive plus mean residual, singles-only models, the non-additive and top-gene coverage metrics, the pair acquisition pool and the discovery analysis were added **after** seeing first results on these same folds, to test whether the first apparent gain was real. All variants are kept in these tables and none was removed, but these folds are not an untouched test set, so a new dataset is needed for confirmation. Nothing here has been validated on an independent dataset or prospectively.
